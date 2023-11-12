package my.aplication.manejointeligentedepragas;

import android.Manifest;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.drawable.ColorDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;

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
//import android.support.v7.widget.LinearLayoutManager;
import androidx.recyclerview.widget.LinearLayoutManager;
//import android.support.v7.widget.RecyclerView;
import androidx.recyclerview.widget.RecyclerView;

//import android.support.v4.app.ActivityCompat;
import androidx.core.app.ActivityCompat;
//import android.support.v4.content.FileProvider;
import androidx.core.content.FileProvider;


import android.os.Bundle;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
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
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;

import com.example.manejointeligentedepragas.BuildConfig;
import com.example.manejointeligentedepragas.R;

import my.aplication.manejointeligentedepragas.RecyclerViewAdapter.AplicacoesRealizadasAdapter;
import my.aplication.manejointeligentedepragas.model.AplicacaoModel;

import com.itextpdf.text.BaseColor;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Element;
import com.itextpdf.text.Font;
import com.itextpdf.text.PageSize;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfPCell;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class RelatorioAplicacoesRealizadas extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener{

    int Cod_Propriedade;
    int codCultura;
    int codPraga;
    String nome;
    boolean aplicado;
    String nomePropriedade;
    String nomePraga;
    int Cod_Talhao;
    String NomeTalhao;
    int Cod_Planta;

    private ArrayList<AplicacaoModel> aplicacoes = new ArrayList<>();

    private Dialog mDialog;

    //usados para itext

    final private int REQUEST_CODE_ASK_PERMISSIONS =111;
    private File pdfFile;

    private DrawerLayout drawerLayout;

    SimpleDateFormat formataDataBR = new SimpleDateFormat("dd-MM-yyyy");
    Date data = new Date();



    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_gerar_relatorio, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.icGeraRelatorio:
                GerarRelatorio();
                return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_relatorio_aplicacoes_realizadas);

        openDialog();

        Cod_Propriedade = getIntent().getIntExtra("Cod_Propriedade", 0);
        codCultura = getIntent().getIntExtra("Cod_Cultura", 0);
        nome = getIntent().getStringExtra("NomeCultura");
        aplicado = getIntent().getBooleanExtra("Aplicado", false);
        codPraga = getIntent().getIntExtra("Cod_Praga", 0);
        nomePropriedade = getIntent().getStringExtra("nomePropriedade");
        nomePraga = getIntent().getStringExtra("nomePraga");
        Cod_Talhao = getIntent().getIntExtra("Cod_Talhao", 0);
        NomeTalhao = getIntent().getStringExtra("NomeTalhao");
        Cod_Planta = getIntent().getIntExtra("Cod_Planta",0);

        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_RAR);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_RAR);
        NavigationView navigationView = findViewById(R.id.nav_view_RAR);
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


        setTitle("MIP² | "+ nomePraga);

        resgataDados(Cod_Talhao, codPraga);
    }

    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            super.onBackPressed();
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
                Intent prop= new Intent(this, Propriedades.class);
                startActivity(prop);
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
                SharedPreferences pref = getApplicationContext().getSharedPreferences("myPrefs",MODE_PRIVATE);
                SharedPreferences.Editor editor = pref.edit();
                editor.putBoolean("isIntroOpened",false);
                editor.commit();

                Intent intro = new Intent(this, IntroActivity.class);
                startActivity(intro);
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

    public void resgataDados(final int Cod_Talhao, final int codPraga){
        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            Toast.makeText(this,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();
            mDialog.dismiss();
        }else { // se tem acesso à internet
            String url = "https://mip.software/phpapp/resgataDadosGraphAplicacao.php?Cod_Talhao="+Cod_Talhao+"&&Cod_Praga="+codPraga;
            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        String Autor;
                        //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i< array.length(); i++){
                            JSONObject obj = array.getJSONObject(i);
                            AplicacaoModel am = new AplicacaoModel();
                            String Data = obj.getString("DataAplicacao");
                            String ano = Data.substring(0,4);
                            String mes = Data.substring(5,7);
                            String dia = Data.substring(8,10);
                            String DataFinal = dia+"-"+mes+"-"+ano;
                            am.setData(DataFinal);
                            am.setDataPlano(obj.getString("DataPlano"));
                            am.setPopPragas(obj.getInt("popPragas"));
                            am.setNumPlantas(obj.getInt("numPlantas"));
                            am.setAutor(obj.getString("Autor"));
                            am.setMetodoAplicado(obj.getString("Metodo"));
                            aplicacoes.add(am);
                        }if(aplicacoes.isEmpty()){
                            AlertDialog.Builder dlgBox = new AlertDialog.Builder(RelatorioAplicacoesRealizadas.this);
                            dlgBox.setCancelable(false);
                            dlgBox.setTitle("Aviso!");
                            dlgBox.setMessage("Você não aplicou nenhum método para esta praga, deseja realizar um plano de amostragem para verificar a necessidade?");
                            dlgBox.setPositiveButton("Sim", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    Intent i = new Intent(RelatorioAplicacoesRealizadas.this, PlanoDeAmostragem.class);
                                    i.putExtra("Cod_Propriedade", Cod_Propriedade);
                                    i.putExtra("Cod_Cultura", codCultura);
                                    i.putExtra("NomeCultura", nome);
                                    i.putExtra("nomePraga", nomePraga);
                                    i.putExtra("Cod_Praga", codPraga);
                                    i.putExtra("Aplicado", aplicado);
                                    i.putExtra("nomePropriedade", nomePropriedade);
                                    i.putExtra("Cod_Talhao", Cod_Talhao);
                                    i.putExtra("NomeTalhao", NomeTalhao);
                                    i.putExtra("Cod_Planta", Cod_Planta);
                                    startActivity(i);
                                }
                            });
                            dlgBox.setNegativeButton("Não", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    Intent i = new Intent(RelatorioAplicacoesRealizadas.this, AcoesCultura.class);
                                    i.putExtra("Cod_Cultura", codCultura);
                                    i.putExtra("NomeCultura", nome);
                                    i.putExtra("Cod_Propriedade", Cod_Propriedade);
                                    i.putExtra("Aplicado", aplicado);
                                    i.putExtra("nomePropriedade", nomePropriedade);
                                    i.putExtra("Cod_Talhao", Cod_Talhao);
                                    i.putExtra("NomeTalhao", NomeTalhao);
                                    i.putExtra("Cod_Planta", Cod_Planta);
                                    startActivity(i);
                                }
                            });
                            dlgBox.show();
                        }else{
                            iniciarRecyclerView();
                        }
                        mDialog.dismiss();

                    } catch (JSONException e) {
                        mDialog.dismiss();
                        Toast.makeText(RelatorioAplicacoesRealizadas.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    mDialog.dismiss();
                    Toast.makeText(RelatorioAplicacoesRealizadas.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));

        }
    }

    private void iniciarRecyclerView(){
        RecyclerView rv = findViewById(R.id.rvAplicacoesRealizadas);
        AplicacoesRealizadasAdapter adapter = new AplicacoesRealizadasAdapter(this, aplicacoes, nome,NomeTalhao);
        rv.setAdapter(adapter);
        rv.setLayoutManager(new LinearLayoutManager(this));
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

    public void GerarRelatorio(){
        //Toast.makeText(RelatorioAplicacoesRealizadas.this, "click", Toast.LENGTH_LONG).show();

        try{
            createPdfWrapper();
        }catch(FileNotFoundException e){
            e.printStackTrace();
            Toast.makeText(RelatorioAplicacoesRealizadas.this, e.toString(), Toast.LENGTH_LONG).show();
        }catch (DocumentException e){
            e.printStackTrace();
            Toast.makeText(RelatorioAplicacoesRealizadas.this, e.toString(), Toast.LENGTH_LONG).show();
        }
    }

    private void createPdfWrapper() throws  FileNotFoundException, DocumentException{

        int hasWriteStoragePermission = ActivityCompat.checkSelfPermission(RelatorioAplicacoesRealizadas.this, Manifest.permission.WRITE_EXTERNAL_STORAGE);

        if(hasWriteStoragePermission != PackageManager.PERMISSION_GRANTED){

            if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
                if(!shouldShowRequestPermissionRationale(Manifest.permission.WRITE_CONTACTS)) {
                    showMessageOKCancel("Para salvar relatórios pela primeira vez, você precisa conceder permissão para armazenamento. Relatórios serão armazenados na pasta MIP, localizada na memória interna do celular. Deseja fazer isso?",
                            new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                                        requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                                                REQUEST_CODE_ASK_PERMISSIONS);
                                    }
                                }
                            });
                    return;
                }
                requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                        REQUEST_CODE_ASK_PERMISSIONS);
            }
            return;
        }else{
            createPdf();
        }


    }

    private void showMessageOKCancel(String Message, DialogInterface.OnClickListener okListener){
            new AlertDialog.Builder(RelatorioAplicacoesRealizadas.this).setMessage(Message)
                    .setPositiveButton("Sim", okListener)
                    .setNegativeButton("Cancelar", null)
                    .create().show();
    }

    private void createPdf() throws FileNotFoundException, DocumentException {
        File docsFolder = new File(Environment.getExternalStorageDirectory() + "/mip/Relatórios de aplicação/"+nome);
        if (!docsFolder.exists()) {
            docsFolder.mkdir();
        }

        String pdfname = nomePropriedade+", "+nome+", "+NomeTalhao+", "+nomePraga+", data: "+formataDataBR.format(data)+".pdf";
        pdfFile = new File(docsFolder.getAbsolutePath(), pdfname);
        OutputStream output = new FileOutputStream(pdfFile);
        Document document = new Document(PageSize.A4);
        PdfPTable table = new PdfPTable(new float[]{3, 3, 3, 3, 3});
        table.getDefaultCell().setHorizontalAlignment(Element.ALIGN_CENTER);
        table.getDefaultCell().setFixedHeight(50);
        table.setTotalWidth(PageSize.A4.getWidth());
        table.setWidthPercentage(100);
        table.getDefaultCell().setVerticalAlignment(Element.ALIGN_MIDDLE);
        table.addCell("Autor");
        table.addCell("Método de controle");
        table.addCell("Data da aplicação");
        table.addCell("Plantas infestadas");
        table.addCell("Número de amostras");
        table.setHeaderRows(1);

        PdfPCell[] cells = table.getRow(0).getCells();

        for (int j = 0; j < cells.length; j++) {
            cells[j].setBackgroundColor(BaseColor.LIGHT_GRAY);
        }

        for (int i = 0; i < aplicacoes.size(); i++) {
            String autor = aplicacoes.get(i).getAutor();
            String metodo = aplicacoes.get(i).getMetodoAplicado();
            String data = aplicacoes.get(i).getData();
            Integer infestacao = aplicacoes.get(i).getPopPragas();
            Integer amostra = aplicacoes.get(i).getNumPlantas();

            table.addCell(String.valueOf(autor));
            table.addCell(String.valueOf(metodo));
            table.addCell(String.valueOf(data));
            table.addCell(String.valueOf(infestacao));
            table.addCell(String.valueOf(amostra));

        }

        PdfWriter.getInstance(document, output);
        document.open();
        Font f = new Font(Font.FontFamily.TIMES_ROMAN, 30.0f, Font.NORMAL, BaseColor.BLACK);
        Font g = new Font(Font.FontFamily.TIMES_ROMAN, 18.0f, Font.NORMAL, BaseColor.BLACK);
        Paragraph inicial = new Paragraph("Relatório de aplicações realizadas\n\n", f);
        inicial.setAlignment(Element.ALIGN_CENTER);
        document.add(inicial);
        document.add(new Paragraph("Propriedade: "+ nomePropriedade +"\n", g));
        document.add(new Paragraph("Cultura: "+ nome +"\n", g));
        document.add(new Paragraph("Talhão: "+NomeTalhao+"\n", g));
        document.add(new Paragraph("Praga: "+ nomePraga +"\n\n", g));
        //document.add(new Paragraph("Pdf File Through Itext", g));
        document.add(table);

        document.close();
        previewPdf();
    }

    private void previewPdf(){
        PackageManager packageManager = RelatorioAplicacoesRealizadas.this.getPackageManager();
        Intent testIntent = new Intent(Intent.ACTION_VIEW);
        testIntent.setType("application/pdf");
        List list = packageManager.queryIntentActivities(testIntent, PackageManager.MATCH_DEFAULT_ONLY);
        if (list.size() > 0) {
            Intent intent = new Intent();
            intent.setAction(Intent.ACTION_VIEW);
            //Uri uri = Uri.fromFile(pdfFile);
            Uri uri = FileProvider.getUriForFile(RelatorioAplicacoesRealizadas.this,  BuildConfig.APPLICATION_ID + ".provider", pdfFile);
            intent.setDataAndType(uri, "application/pdf");
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            RelatorioAplicacoesRealizadas.this.startActivity(intent);
        } else {
            Toast.makeText(RelatorioAplicacoesRealizadas.this, "Você não possui um aplicativo para abrir o PDF, baixe um para abrir o relatório.", Toast.LENGTH_SHORT).show();
        }

    }

}
