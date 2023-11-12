package my.aplication.manejointeligentedepragas;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.drawable.ColorDrawable;

//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
//import android.support.design.widget.NavigationView;
import com.google.android.material.navigation.NavigationView;
//import android.support.v4.view.GravityCompat;
import androidx.core.view.GravityCompat;
//import android.support.v4.widget.DrawerLayout;
import androidx.drawerlayout.widget.DrawerLayout;
//import android.support.v7.app.ActionBarDrawerToggle;
import androidx.appcompat.app.ActionBarDrawerToggle;
//import android.support.v7.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatActivity;
//import android.support.v7.widget.Toolbar;
import androidx.appcompat.widget.Toolbar;

//import android.support.v4.view.ViewPager;
import androidx.viewpager.widget.ViewPager;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Atinge;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_PlanoAmostragem;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Praga;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_UltimosPlanos;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;
import my.aplication.manejointeligentedepragas.ImageAdapter.ViewPagerAdapter;

import com.example.manejointeligentedepragas.R;

import my.aplication.manejointeligentedepragas.model.AtingeModel;
import my.aplication.manejointeligentedepragas.model.PlanoAmostragemModel;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;


public class PlanoDeAmostragem extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener{

    private static final String TAG = "PlanoDeAmostragem";

    ArrayList<String> urlsPragas = new ArrayList<>();

    int codPropriedade;
    int codPraga;
    int codCultura;
    String nome;
    String nomePraga;
    boolean aplicado;
    String nomePropriedade;
    int numAmostras;
    int Cod_Talhao;
    String NomeTalhao;
    int Cod_Planta;

    int plantasTalhao;
    int pontosTalhao;
    int plantaPonto;

    double nivelControle;

    TextView tvplantasTalhao;
    TextView tvpontosTalhao;
    TextView tvplantaPonto;
    TextView tvContagemPlantas;
    TextView tvNumAmostras;
    TextView tvContagemNumAmostras;

    Button presencaPraga;
    Button ausenciaPraga;

    Button btnFinalizaPlano;

    Button btnCorrigir;
    Boolean ultimoClick = false;

    ArrayList<Integer> codTalhoes = new ArrayList<Integer>();

    //contadores
    int countInfestacao=0;
    int countPlantas=1;
    int contAmostra=1;
    int countNumeroAmostras;

    //data
    SimpleDateFormat formataData = new SimpleDateFormat("yyyy-MM-dd");
    Date data = new Date();
    String dataFormatada = formataData.format(data);

    //controla ou não controla
    Boolean controla = false;

    ImageView imgInfo;
    String amostra;

    //data ultimo plano
    String ultimoPlano;

    private Dialog mDialog;


    private DrawerLayout drawerLayout;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_plano_de_amostragem);
        openDialog();

        codPropriedade = getIntent().getIntExtra("Cod_Propriedade", 0);
        codCultura = getIntent().getIntExtra("Cod_Cultura", 0);
        nome = getIntent().getStringExtra("NomeCultura");
        nomePraga = getIntent().getStringExtra("nomePraga");
        codPraga = getIntent().getIntExtra("Cod_Praga", 0);
        aplicado = getIntent().getBooleanExtra("Aplicado", false);
        nomePropriedade = getIntent().getStringExtra("nomePropriedade");
        Cod_Talhao = getIntent().getIntExtra("Cod_Talhao", 0);
        NomeTalhao = getIntent().getStringExtra("NomeTalhao");
        Cod_Planta = getIntent().getIntExtra("Cod_Planta",0);


        tvplantasTalhao = findViewById(R.id.tvNPlantasPorTalhao);
        tvplantaPonto = findViewById(R.id.tvNPlantasPorPonto);
        tvpontosTalhao = findViewById(R.id.tvNPontosPorTalhao);
        tvNumAmostras = findViewById(R.id.tvNumAostras);

        tvContagemPlantas = findViewById(R.id.tvContagemPlantas);
        tvContagemNumAmostras = findViewById(R.id.tvContagemAmostras);

        presencaPraga = findViewById(R.id.btnPresencaPraga);
        ausenciaPraga = findViewById(R.id.btnAusenciaPraga);
        imgInfo = findViewById(R.id.ImgInfo);

        setTitle("MIP² | "+nomePraga);

        ResgatarDataUltimoPlano(codPraga,Cod_Talhao);
        ChamaAmostra(codPraga);
        ResgatarAtinge(codPraga,codCultura);

        tvContagemPlantas.setText(String.valueOf(countPlantas));
        tvContagemNumAmostras.setText(String.valueOf(contAmostra));

        btnFinalizaPlano = findViewById(R.id.btnFinalizarPlano);
        btnCorrigir = findViewById(R.id.btnCorrige);

        btnCorrigir.setVisibility(View.INVISIBLE);

        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_plano_de_amostragem);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_plano_de_amostragem);
        NavigationView navigationView = findViewById(R.id.nav_view_plano_de_amostragem);
        navigationView.setNavigationItemSelectedListener(this);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawerLayout.addDrawerListener(toggle);
        toggle.syncState();
        View headerView = navigationView.getHeaderView(0);

        Controller_Usuario controller_usuario = new Controller_Usuario(getBaseContext());
        String nomeUsu = controller_usuario.getUser().getNome();
        String emailUsu = controller_usuario.getUser().getEmail();

        TextView nomeMenu = headerView.findViewById(R.id.nomeMenu);
        nomeMenu.setText(nomeUsu);

        TextView emailMenu = headerView.findViewById(R.id.emailMenu);
        emailMenu.setText(emailUsu);



        ausenciaPraga.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(numAmostras == 1){
                    ultimoClick = false; // saber qual click foi para a correção
                    countPlantas++;
                    tvContagemPlantas.setText(String.valueOf(countPlantas));
                    checarTalhao();
                }else{
                    if(contAmostra == numAmostras){
                        ultimoClick = false; // saber qual click foi para a correção
                        contAmostra = 1;
                        countPlantas++;
                        countNumeroAmostras ++;
                        tvContagemPlantas.setText(String.valueOf(countPlantas));
                        tvContagemNumAmostras.setText(String.valueOf(contAmostra));
                        checarTalhao();
                    }else{
                        ultimoClick = false; // saber qual click foi para a correção
                        contAmostra ++;
                        countNumeroAmostras ++;
                        tvContagemNumAmostras.setText(String.valueOf(contAmostra));
                        checarTalhao();
                    }

                }
            }
        });
        presencaPraga.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(numAmostras == 1){
                    ultimoClick = true; // saber qual click foi para a correção
                    countInfestacao++;
                    countPlantas++;
                    tvContagemPlantas.setText(String.valueOf(countPlantas));
                    checarTalhao();
                }else{
                    if(contAmostra == numAmostras){
                        ultimoClick = true; // saber qual click foi para a correção
                        contAmostra = 1;
                        countPlantas++;
                        countInfestacao++;
                        countNumeroAmostras++;
                        tvContagemPlantas.setText(String.valueOf(countPlantas));
                        tvContagemNumAmostras.setText(String.valueOf(contAmostra));
                        checarTalhao();
                    }else{
                        ultimoClick = true; // saber qual click foi para a correção
                        contAmostra ++;
                        countInfestacao++;
                        countNumeroAmostras++;
                        tvContagemNumAmostras.setText(String.valueOf(contAmostra));
                        checarTalhao();
                    }
                }
            }
        });

        /*btnTalhaoAnterior.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                exibirCaixaDialogoAnterior();
            }
        });*/

        btnFinalizaPlano.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                exibirCaixaDialogoProximo();
            }
        });

        btnCorrigir.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(numAmostras == 1){
                    countPlantas--;
                    if(ultimoClick){
                        countInfestacao--;
                    }
                    tvContagemPlantas.setText(String.valueOf(countPlantas));
                    btnCorrigir.setVisibility(View.INVISIBLE);
                }else{
                    if(contAmostra == 1){
                        countPlantas--;
                        countNumeroAmostras--;
                        contAmostra = numAmostras;
                        if(ultimoClick){
                            countInfestacao--;
                        }
                        tvContagemPlantas.setText(String.valueOf(countPlantas));
                        tvContagemNumAmostras.setText(String.valueOf(contAmostra));
                        btnCorrigir.setVisibility(View.INVISIBLE);
                    }else{
                        contAmostra --;
                        countNumeroAmostras--;
                        if(ultimoClick){
                            countInfestacao--;
                        }
                        tvContagemNumAmostras.setText(String.valueOf(contAmostra));
                        btnCorrigir.setVisibility(View.INVISIBLE);
                    }
                }

            }
        });

        btnCorrigir.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {
                exibirCaixaDialogoOnLongClickCorrigir();
                return true;
            }
        });


        imgInfo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ExibeAmostra();
            }
        });
        //click listener do talhao anterior: diminuir count talhão, zerar count planta e apagar ultima posição do arraylist de planos
        //click listener do proximo talhao: aumentar count talhão, zerar count planta , ver se é o ultimo talhao,
        // se for o ultimo talhao +1 salvar no arraylist e passa pra proxima intent

        //mensagem em ambos botoes

        //if na função pra checar se o count de talhao é maior que o numero de talhoes da cultura, se for salva o arraylist de planos
        // no banco online

        // booleano no PlanoDeAmostragemModel ou na classe e mandar pra proxima intent para verificar se controla ou nao

        // tem public void onClick(View v) {} para detectar quando clica no botão


    }

    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            exibirCaixaDialogoOnbackPressed();
        }
    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
        switch (menuItem.getItemId()){
            case R.id.drawerPerfil:
                Intent i= new Intent(this, Perfil.class);
                startActivity(i);
                break;
            case R.id.drawerProp:
                exibirCaixaDialogoOnPropPressed();
                break;

            case R.id.drawerPlantas:
                Intent j = new Intent(this, VisualizaPlantas.class);
                startActivity(j);
                break;

            case R.id.drawerPrag:
                Intent k = new Intent(this, VisualizaPragas.class);
                startActivity(k);
                break;

            case R.id.drawerMet:
                Intent l = new Intent(this, VisualizaMetodos.class);
                startActivity(l);
                break;

            case R.id.drawerSobreMip:
                Intent p = new Intent(this, SobreMIP.class);
                startActivity(p);
                break;

            case R.id.drawerTutorial:
                exibirCaixaDialogoOnMenuPressed();
                break;

            case R.id.drawerSobre:
                Intent pp = new Intent(this, Sobre.class);
                startActivity(pp);
                break;

            case R.id.drawerReferencias:
                Intent pi = new Intent(this, Referencias.class);
                startActivity(pi);
                break;

            case R.id.drawerRecomendações:
                Intent pa = new Intent(this, RecomendacoesMAPA.class);
                startActivity(pa);
                break;
        }
        drawerLayout.closeDrawer(GravityCompat.START);
        return true;
    }






    public void ResgatarAtinge(int codPraga, final int codCultura){
        Controller_Atinge ca = new Controller_Atinge(PlanoDeAmostragem.this);
        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            AtingeModel atinge = ca.getAtinge(codPraga,Cod_Planta);
            plantasTalhao = atinge.getNumeroPlantasAmostradas();
            pontosTalhao = atinge.getPontosPorTalhao();
            plantaPonto = atinge.getPlantasPorPonto();
            nivelControle = atinge.getNivelDeControle();
            numAmostras =  atinge.getNumAmostras();

            if(numAmostras!=1){
                tvContagemNumAmostras.setVisibility(View.VISIBLE);
                tvNumAmostras.setVisibility(View.VISIBLE);
                tvplantasTalhao.setText(String.valueOf(plantasTalhao));
                tvplantaPonto.setText(String.valueOf(plantaPonto));
                tvpontosTalhao.setText(String.valueOf(pontosTalhao));
            }else{
                tvplantasTalhao.setText(String.valueOf(plantasTalhao));
                tvplantaPonto.setText(String.valueOf(plantaPonto));
                tvpontosTalhao.setText(String.valueOf(pontosTalhao));
            }

        }else { // se tem acesso à internet
            String url = "https://mip.software/phpapp/resgatarAtinge.php?Cod_Cultura="+codCultura+"&&Cod_Praga="+codPraga;


            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i< array.length(); i++){
                            JSONObject obj = array.getJSONObject(i);
                            plantasTalhao = obj.getInt("NumeroPlantasAmostradas");
                            pontosTalhao = obj.getInt("PontosPorTalhao");
                            plantaPonto = obj.getInt("PlantasPorPonto");
                            nivelControle = obj.getDouble("NivelDeControle");
                            numAmostras = obj.getInt("NumAmostras");
                        }
                        if(numAmostras!=1){
                            tvContagemNumAmostras.setVisibility(View.VISIBLE);
                            tvNumAmostras.setVisibility(View.VISIBLE);
                            tvplantasTalhao.setText(String.valueOf(plantasTalhao));
                            tvplantaPonto.setText(String.valueOf(plantaPonto));
                            tvpontosTalhao.setText(String.valueOf(pontosTalhao));
                        }else{
                            tvplantasTalhao.setText(String.valueOf(plantasTalhao));
                            tvplantaPonto.setText(String.valueOf(plantaPonto));
                            tvpontosTalhao.setText(String.valueOf(pontosTalhao));
                        }


                        //ResgatarTalhao(codCultura);
                        mDialog.dismiss();

                    } catch (JSONException e) {
                        mDialog.dismiss();
                        Toast.makeText(PlanoDeAmostragem.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    mDialog.dismiss();
                    Toast.makeText(PlanoDeAmostragem.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));

        }
    }

    /*public void ResgatarTalhao(int codCultura){
        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            mDialog.dismiss();
            Toast.makeText(this,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();
        }else { // se tem acesso à internet
            String url = "http://mip2.000webhostapp.com/resgatarTalhao.php?Cod_Cultura="+codCultura;
            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i< array.length(); i++){
                            JSONObject obj = array.getJSONObject(i);
                            codTalhoes.add(obj.getInt("Cod_Talhao"));
                        }
                        if(codTalhoes.size()==1) {
                            btnTalhaoSuperior.setText("Finalizar");
                        }
                    } catch (JSONException e) {
                        mDialog.dismiss();
                        Toast.makeText(PlanoDeAmostragem.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    mDialog.dismiss();
                    Toast.makeText(PlanoDeAmostragem.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));

        }
    }
    */

    public void checarTalhao(){
        btnCorrigir.setVisibility(View.VISIBLE);
        if(countPlantas == plantasTalhao+1){
            presencaPraga.setVisibility(View.INVISIBLE);
            ausenciaPraga.setVisibility(View.INVISIBLE);
            btnCorrigir.setVisibility(View.INVISIBLE);
            btnFinalizaPlano.setVisibility(View.INVISIBLE);
            if(numAmostras == 1){
                PlanoAmostragemModel pa = new PlanoAmostragemModel(dataFormatada,countInfestacao,countPlantas-1,Cod_Talhao,codPraga);
                //countTalhao++;
                //btnTalhaoAnterior.setVisibility(View.VISIBLE);
                SalvarPlanoAmostragem(pa);
                btnCorrigir.setVisibility(View.INVISIBLE);
            }else{
                PlanoAmostragemModel pa = new PlanoAmostragemModel(dataFormatada,countInfestacao,countNumeroAmostras,Cod_Talhao,codPraga);
                //countTalhao++;
                //btnTalhaoAnterior.setVisibility(View.VISIBLE);
                SalvarPlanoAmostragem(pa);
                btnCorrigir.setVisibility(View.INVISIBLE);
            }
        }
    }

    public void exibirCaixaDialogoProximo()
    {
        if(countPlantas == 1){
            AlertDialog.Builder dlgBox = new AlertDialog.Builder(this);
            dlgBox.setTitle("Aviso!");
            dlgBox.setMessage("Por favor, faça a amostragem de pelo menos uma planta.");
            dlgBox.setPositiveButton("Ok", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    //não faz nada
                }
            });

            dlgBox.show();
        }else {
            AlertDialog.Builder dlgBox = new AlertDialog.Builder(this);
            dlgBox.setTitle("Aviso!");
            dlgBox.setMessage("Você não verificou todas as plantas necessárias e isso afetará a precisão do plano de amostragem, deseja avançar?");
            dlgBox.setPositiveButton("Sim", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    if(numAmostras == 1){
                        PlanoAmostragemModel pa = new PlanoAmostragemModel(dataFormatada, countInfestacao, countPlantas - 1, Cod_Talhao, codPraga);
                        SalvarPlanoAmostragem(pa);
                        btnCorrigir.setVisibility(View.INVISIBLE);
                    }else{
                        PlanoAmostragemModel pa = new PlanoAmostragemModel(dataFormatada, countInfestacao, countNumeroAmostras, Cod_Talhao, codPraga);
                        SalvarPlanoAmostragem(pa);
                        btnCorrigir.setVisibility(View.INVISIBLE);
                    }
                }
            });

            dlgBox.setNegativeButton("Não", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    // não faz nada
                }
            });

            dlgBox.show();
        }
    }

    /*public void exibirCaixaDialogoAnterior()
    {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(this);
        dlgBox.setTitle("Aviso!");
        dlgBox.setMessage("Deseja refazer o plano de amostragem do talhão anterior?");
        dlgBox.setPositiveButton("Sim", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                btnCorrigir.setVisibility(View.INVISIBLE);
                if(countTalhao != 1) {
                    planos.remove(countTalhao - 2);
                    countTalhao--;
                    btnTalhaoSuperior.setText("Próximo talhão");
                    if(countTalhao == 1){
                        btnTalhaoAnterior.setVisibility(View.INVISIBLE);
                    }
                    contAmostra = 1;
                    countPlantas = 1;
                    countInfestacao = 0;
                    tvContagemNumAmostras.setText(String.valueOf(contAmostra));
                    tvContagemPlantas.setText(String.valueOf(countPlantas));
                    tvContagemTalhoes.setText(String.valueOf(countTalhao));
                }
            }
        });

        dlgBox.setNegativeButton("Não", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                // não faz nada
            }
        });

        dlgBox.show();

    }*/

    public void exibirCaixaDialogoOnbackPressed()
    {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(this);
        dlgBox.setTitle("ATENÇÃO!");
        dlgBox.setMessage("Deseja cancelar o plano de amostragem?");
        dlgBox.setPositiveButton("Sim", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                Intent i = new Intent(PlanoDeAmostragem.this, AcoesCultura.class);
                i.putExtra("Cod_Talhao", Cod_Talhao);
                i.putExtra("NomeTalhao", NomeTalhao);
                i.putExtra("Cod_Propriedade", codPropriedade);
                i.putExtra("Cod_Cultura", codCultura);
                i.putExtra("NomeCultura", nome);
                i.putExtra("Aplicado", aplicado);
                i.putExtra("nomePropriedade", nomePropriedade);
                i.putExtra("Cod_Planta", Cod_Planta);
                startActivity(i);
            }
        });

        dlgBox.setNegativeButton("Não", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                // não faz nada
            }
        });

        dlgBox.show();

    }

    public void exibirCaixaDialogoOnMenuPressed()
    {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(this);
        dlgBox.setTitle("ATENÇÃO!");
        dlgBox.setMessage("Caso abra o tutotial o plano de amostragem será cancelado, deseja fazer isso?");
        dlgBox.setPositiveButton("Sim", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                SharedPreferences pref = getApplicationContext().getSharedPreferences("myPrefs",MODE_PRIVATE);
                SharedPreferences.Editor editor = pref.edit();
                editor.putBoolean("isIntroOpened",false);
                editor.commit();

                Intent intro = new Intent(PlanoDeAmostragem.this, IntroActivity.class);
                startActivity(intro);
            }
        });

        dlgBox.setNegativeButton("Não", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                // não faz nada
            }
        });

        dlgBox.show();

    }

    public void exibirCaixaDialogoOnPropPressed()
    {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(this);
        dlgBox.setTitle("ATENÇÃO!");
        dlgBox.setMessage("Caso vá para a tela de propriedades o plano de amostragem será cancelado, deseja fazer isso?");
        dlgBox.setPositiveButton("Sim", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                Intent prop= new Intent(PlanoDeAmostragem.this, Propriedades.class);
                startActivity(prop);
            }
        });

        dlgBox.setNegativeButton("Não", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                // não faz nada
            }
        });

        dlgBox.show();

    }

    public void exibirCaixaDialogoOnLongClickCorrigir()
    {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(this);
        dlgBox.setTitle("Aviso!");
        dlgBox.setMessage("Deseja refazer a contagem deste talhão?");
        dlgBox.setPositiveButton("Sim", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                countPlantas = 1;
                countInfestacao = 0;
                contAmostra = 1;
                tvContagemPlantas.setText(String.valueOf(countPlantas));
                tvContagemNumAmostras.setText(String.valueOf(contAmostra));
                btnCorrigir.setVisibility(View.INVISIBLE);
            }
        });

        dlgBox.setNegativeButton("Não", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                // não faz nada
            }
        });

        dlgBox.show();

    }

    public void SalvarPlanoAmostragem(PlanoAmostragemModel pa){
        //verifica todos os talhões, se qualquer um tiver o nível de controle atingido, é necessário o controle em toda cultura
                // cálculo para conferir o nível de controle
        Controller_PlanoAmostragem cpa = new Controller_PlanoAmostragem(PlanoDeAmostragem.this);
        final Controller_UltimosPlanos cup = new Controller_UltimosPlanos(PlanoDeAmostragem.this);
        if(numAmostras == 1){
            int ci = countInfestacao;
            int cp = countPlantas;
            double dCI;
            double dCP;
            dCI = (double)ci;
            dCP = (double)cp;
            double porcentagemInfestadas = (dCI/ dCP);
            //Toast.makeText(PlanoDeAmostragem.this, ""+porcentagemInfestadas,Toast.LENGTH_LONG).show();
            if(porcentagemInfestadas >= nivelControle){
                controla = true;
            }
        }else{
            int ci = countInfestacao;
            int cp = countNumeroAmostras;
            double dCI;
            double dCP;
            dCI = (double)ci;
            dCP = (double)cp;
            double porcentagemInfestadas = (dCI/ dCP);
            //Toast.makeText(PlanoDeAmostragem.this, ""+porcentagemInfestadas,Toast.LENGTH_LONG).show();
            if(porcentagemInfestadas >= nivelControle){
                controla = true;
            }
        }

            Utils u = new Utils();
            if(!u.isConected(getBaseContext()))
            {
                mDialog.dismiss();
                pa.setSync_Status(1);
                cpa.addPlano(pa);
                cup.addUltimosPlanos(dataFormatada,Cod_Talhao,codPraga);


                Intent i = new Intent(PlanoDeAmostragem.this, ConfirmacaoPlanoAmostragem.class);
                i.putExtra("Cod_Propriedade", codPropriedade);
                i.putExtra("Cod_Cultura", codCultura);
                i.putExtra("NomeCultura", nome);
                i.putExtra("NomePraga",nomePraga);
                i.putExtra("Cod_Praga",codPraga);
                i.putExtra("Controla",controla);
                i.putExtra("Aplicado", aplicado);
                i.putExtra("nomePropriedade", nomePropriedade);
                i.putExtra("Cod_Talhao", Cod_Talhao);
                i.putExtra("NomeTalhao", NomeTalhao);
                i.putExtra("Cod_Planta", Cod_Planta);
                startActivity(i);
            }else { // se tem acesso à internet
                Controller_Usuario cu = new Controller_Usuario(getBaseContext());
                String Autor = cu.getUser().getNome();

                String url = "https://mip.software/phpapp/salvaPlanoAmostragem.php?Cod_Talhao=" + Cod_Talhao
                        +"&&Data="+dataFormatada
                        +"&&PlantasInfestadas="+pa.getPlantasInfestadas()
                        +"&&PlantasAmostradas="+pa.getPlantasAmostradas()
                        +"&&Cod_Praga="+codPraga
                        +"&&Autor="+Autor;

                RequestQueue queue = Volley.newRequestQueue(this);
                queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                    @Override
                    public void onResponse(String response) {
                        //Parsing json
                        //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                        try {

                            JSONObject obj1 = new JSONObject(response);
                            boolean confirmacao = obj1.getBoolean("confirmacao");
                            if(confirmacao){
                                Intent i = new Intent(PlanoDeAmostragem.this, ConfirmacaoPlanoAmostragem.class);
                                i.putExtra("Cod_Propriedade", codPropriedade);
                                i.putExtra("Cod_Cultura", codCultura);
                                i.putExtra("NomeCultura", nome);
                                i.putExtra("NomePraga",nomePraga);
                                i.putExtra("Cod_Praga",codPraga);
                                i.putExtra("Controla",controla);
                                i.putExtra("Aplicado", aplicado);
                                i.putExtra("nomePropriedade", nomePropriedade);
                                i.putExtra("Cod_Talhao", Cod_Talhao);
                                i.putExtra("NomeTalhao", NomeTalhao);
                                i.putExtra("Cod_Planta", Cod_Planta);
                                startActivity(i);
                            }else{
                                mDialog.dismiss();
                                Toast.makeText(PlanoDeAmostragem.this, "Plano de amostragem não cadastrado! Tente novamente",Toast.LENGTH_LONG).show();
                            }
                        } catch (JSONException e) {
                            mDialog.dismiss();
                            Toast.makeText(PlanoDeAmostragem.this, e.toString(), Toast.LENGTH_LONG).show();
                        }
                    }
                }, new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        mDialog.dismiss();
                        Toast.makeText(PlanoDeAmostragem.this,error.toString(), Toast.LENGTH_LONG).show();
                    }
                }));

            }
    }

    public void ChamaAmostra(int codPraga){
        Utils u = new Utils();
        Controller_Praga cp = new Controller_Praga(PlanoDeAmostragem.this);
        if(!u.isConected(getBaseContext()))
        {
            amostra = cp.getAmostra(codPraga);

        }else { // se tem acesso à internet
            String url = "https://mip.software/phpapp/ChamaAmostra.php?Cod_Praga=" + codPraga;

            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i< array.length(); i++) {
                            JSONObject obj = array.getJSONObject(i);
                            amostra = obj.getString("Localizacao");
                        }
                    } catch (JSONException e) {
                        mDialog.dismiss();
                        Toast.makeText(PlanoDeAmostragem.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    mDialog.dismiss();
                    Toast.makeText(PlanoDeAmostragem.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));
        }
    }

    public void ResgatarFotoAmostra(final ViewPager viewPager, int codP){
        Utils u = new Utils();
        if (!u.isConected(getBaseContext())) {
            mDialog.dismiss();
            //Toast.makeText(this, "Habilite a conexão com a internet", Toast.LENGTH_LONG).show();
        } else { // se tem acesso à internet

            String url = "https://mip.software/phpapp/resgatarFotoAmostra.php?Cod_Praga="+codP;

            RequestQueue queue = Volley.newRequestQueue(PlanoDeAmostragem.this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i < array.length(); i++) {
                            JSONObject obj = array.getJSONObject(i);
                            urlsPragas.add("https://mip.software/amostras/"+obj.getString("FotoAmostra"));
                        }
                        if(urlsPragas.size()>0){
                            ViewPagerAdapter adapterAmostras = new ViewPagerAdapter(PlanoDeAmostragem.this,urlsPragas);
                            viewPager.setAdapter(adapterAmostras);
                        }else{
                            viewPager.setVisibility(View.GONE);
                        }

                    } catch (JSONException e) {
                        mDialog.dismiss();
                        Toast.makeText(PlanoDeAmostragem.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    mDialog.dismiss();
                    Toast.makeText(PlanoDeAmostragem.this, error.toString(), Toast.LENGTH_LONG).show();
                }
            }));
        }
    }

    public void ExibeAmostra(){
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(this);
        dlgBox.setTitle("Amostra: ");

        LayoutInflater inflater= LayoutInflater.from(this);
        View view = inflater.inflate(R.layout.imagem_amostra, null);
        TextView textview = (TextView)view.findViewById(R.id.textmsg);
        textview.setText(amostra+"\n");
        ViewPager viewPager = (ViewPager)view.findViewById(R.id.ViewPagerAmostra);


        Utils u = new Utils();
        Controller_Praga cp = new Controller_Praga(PlanoDeAmostragem.this);
        if(!u.isConected(getBaseContext()))
        {
            viewPager.setVisibility(View.GONE);
        }else {
            ResgatarFotoAmostra(viewPager, codPraga);
        }

        //ImageView showImg = new ImageView(this);
        //ImageView showImg = (ImageView)view.findViewById(R.id.imgView);
        //showImg.setImageResource(R.drawable.mineiro);
        dlgBox.setView(view);
        //dlgBox.setMessage(amostra+"\n");
        dlgBox.setPositiveButton("Ok", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {

            }
        });

        dlgBox.show();
    }

    public void ResgatarDataUltimoPlano(int codPraga, final int codTalhao){
        final Controller_UltimosPlanos cup = new Controller_UltimosPlanos(PlanoDeAmostragem.this);
        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            ultimoPlano = cup.getUltimosPlanos(codTalhao,codPraga);
            mDialog.dismiss();
            if(ultimoPlano!=null) {
                //compare to volta 0 se forem iguais ou numero negativo/positivo se o dia for menor/maior
                if (dataFormatada.equals(ultimoPlano)) {
                    AlertDialog.Builder dlgBox = new AlertDialog.Builder(PlanoDeAmostragem.this);
                    dlgBox.setCancelable(false);
                    dlgBox.setTitle("Aviso!");
                    dlgBox.setMessage("Você já realizou uma contagem dessa praga hoje, por favor espere até amanhã para realizar uma nova contagem.");
                    dlgBox.setPositiveButton("Voltar", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            Intent i = new Intent(PlanoDeAmostragem.this, AcoesCultura.class);
                            i.putExtra("Cod_Talhao", Cod_Talhao);
                            i.putExtra("NomeTalhao", NomeTalhao);
                            i.putExtra("Cod_Propriedade", codPropriedade);
                            i.putExtra("Cod_Cultura", codCultura);
                            i.putExtra("NomeCultura", nome);
                            i.putExtra("Aplicado", aplicado);
                            i.putExtra("nomePropriedade", nomePropriedade);
                            i.putExtra("Cod_Planta", Cod_Planta);
                            startActivity(i);
                        }
                    });
                    dlgBox.show();
                }
            }
            //Toast.makeText(this,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();
        }else { // se tem acesso à internet
            String url = "https://mip.software/phpapp/resgataDataUltimoPlano.php?Cod_Praga=" + codPraga+"&&Cod_Talhao="+codTalhao;

            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i < array.length(); i++) {
                            JSONObject obj = array.getJSONObject(i);
                            ultimoPlano = obj.getString("Data");
                        }
                        if(ultimoPlano!=null) {
                                //compare to volta 0 se forem iguais ou numero negativo/positivo se o dia for menor/maior
                                if (dataFormatada.equals(ultimoPlano)) {
                                    AlertDialog.Builder dlgBox = new AlertDialog.Builder(PlanoDeAmostragem.this);
                                    dlgBox.setCancelable(false);
                                    dlgBox.setTitle("Aviso!");
                                    dlgBox.setMessage("Você já realizou uma contagem dessa praga hoje, por favor espere até amanhã para realizar uma nova contagem.");
                                    dlgBox.setPositiveButton("Voltar", new DialogInterface.OnClickListener() {
                                        @Override
                                        public void onClick(DialogInterface dialog, int which) {
                                            Intent i = new Intent(PlanoDeAmostragem.this, AcoesCultura.class);
                                            i.putExtra("Cod_Talhao", Cod_Talhao);
                                            i.putExtra("NomeTalhao", NomeTalhao);
                                            i.putExtra("Cod_Propriedade", codPropriedade);
                                            i.putExtra("Cod_Cultura", codCultura);
                                            i.putExtra("NomeCultura", nome);
                                            i.putExtra("Aplicado", aplicado);
                                            i.putExtra("nomePropriedade", nomePropriedade);
                                            i.putExtra("Cod_Planta", Cod_Planta);
                                            startActivity(i);
                                        }
                                    });
                                    dlgBox.show();
                                }
                        }
                    } catch (JSONException e) {
                        mDialog.dismiss();
                        Toast.makeText(PlanoDeAmostragem.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    mDialog.dismiss();
                    Toast.makeText(PlanoDeAmostragem.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));
        }
    }



    public void openDialog(){
        mDialog = new Dialog(this);
        //vamos remover o titulo da Dialog
        mDialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        //vamos carregar o xml personalizado
        mDialog.setContentView(R.layout.dialog);
        //DEixamos transparente
        mDialog.getWindow().setBackgroundDrawable(new ColorDrawable(android.graphics.Color.TRANSPARENT));
        // não permitimos fechar esta dialog
        mDialog.setCancelable(false);
        //temos a instancia do ProgressBar!
        final ProgressBar progressBar = ProgressBar.class.cast(mDialog.findViewById(R.id.progressBar));

        mDialog.show();

        // mDialog.dismiss(); -> para fechar a dialog

    }

}
