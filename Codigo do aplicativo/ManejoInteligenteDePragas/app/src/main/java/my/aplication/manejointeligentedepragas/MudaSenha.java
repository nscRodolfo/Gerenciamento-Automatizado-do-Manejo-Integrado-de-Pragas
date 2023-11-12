package my.aplication.manejointeligentedepragas;

import android.content.Intent;
import android.content.SharedPreferences;

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

import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
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

import com.example.manejointeligentedepragas.R;

import my.aplication.manejointeligentedepragas.model.UsuarioModel;

import org.json.JSONException;
import org.json.JSONObject;

import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public class MudaSenha extends AppCompatActivity  implements NavigationView.OnNavigationItemSelectedListener{

    EditText etSenhaAntiga;
    EditText etNovaSenha;
    EditText etConfSenha;

    Button salvar;

    String senha;
    int codigo;
    String nome;
    String tipo;
    String telefone;
    String email;

    private DrawerLayout drawerLayout;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_muda_senha);

        etSenhaAntiga = findViewById(R.id.etSenhaAtual);
        etNovaSenha = findViewById(R.id.etNovaSenha);
        etConfSenha = findViewById(R.id.etConfSenha);

        salvar = findViewById(R.id.btnMudaSenha);


        Controller_Usuario cu1 = new Controller_Usuario(getBaseContext());
        senha = cu1.getUser().getSenha();
        codigo = cu1.getUser().getCod_Usuario();
        nome = cu1.getUser().getNome();
        telefone = cu1.getUser().getTelefone();
        email = cu1.getUser().getEmail();
        tipo = cu1.getUser().getTipo();

        salvar.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Utils u = new Utils();
                if(!u.isConected(getBaseContext()))
                {
                    Toast.makeText(MudaSenha.this,"Habilite a conexão com a internet", Toast.LENGTH_LONG).show();
                }else { // se tem acesso à internet chama logar
                    salvaSenha();
                }
            }
        });

        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_muda);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_muda);
        NavigationView navigationView = findViewById(R.id.nav_view_muda);
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

    }


    public void salvaSenha(){
        if (etSenhaAntiga.toString().isEmpty()){
            etSenhaAntiga.setError("A antiga senha é obrigatória!");
        }else if (!convertPassMd5(etSenhaAntiga.getText().toString()).equals(senha)){
            etSenhaAntiga.setError("A senha informada não é igual a atual!");
        }else if (etNovaSenha.getText().toString().isEmpty()){
            etNovaSenha.setError("Senha é obrigatória!");
        }else if (etConfSenha.getText().toString().isEmpty()){
            etConfSenha.setError("Confirmação de senha é obrigatória!");
        }else if (!etNovaSenha.getText().toString().equals(etConfSenha.getText().toString())){
            etConfSenha.setError("As senhas informadas não são iguais!");
        }else if (etConfSenha.getText().toString().length() < 6){
            etConfSenha.setError("A senha precisa ter mais que 6 digitos");
        }else {
            final String senhaNova = convertPassMd5(etConfSenha.getText().toString());
            final UsuarioModel u = new UsuarioModel();
            u.setCod_Usuario(codigo);
            u.setTipo(tipo);
            u.setTelefone(telefone);
            u.setNome(nome);
            u.setEmail(email);
            u.setSenha(senhaNova);

            Controller_Usuario cu1 = new Controller_Usuario(getBaseContext());
            cu1.updateUser(u);

            String url = "https://mip.software/phpapp/mudaSenha.php?Senha=" + senhaNova + "&&Cod_Usu=" + codigo;
            RequestQueue queue = Volley.newRequestQueue(this);

            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {
                @Override
                public void onResponse(String response) {

                    // Parsing json
                    try {
                        JSONObject Obj = new JSONObject(response);
                        boolean confirmacao = Obj.getBoolean("confirmacao");

                        if (confirmacao) {
                            Toast.makeText(MudaSenha.this, "Atualização realizada com sucesso!", Toast.LENGTH_LONG).show();
                            Controller_Usuario cu1 = new Controller_Usuario(getBaseContext());
                            cu1.updateUser(u);
                            final Intent k = new Intent(MudaSenha.this, Perfil.class);
                            startActivity(k);
                        } else {
                            Toast.makeText(MudaSenha.this, "Atualização de senha falhou, tente novamente mais tarde!", Toast.LENGTH_LONG).show();
                        }

                    } catch (JSONException e) {
                        e.printStackTrace();
                    }

                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(MudaSenha.this, error.toString(), Toast.LENGTH_LONG).show();
                }
            }));
        }
    }

    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            super.onBackPressed();
        }
    }


    public static String convertPassMd5(String pass) {
        String password = null;
        MessageDigest mdEnc;
        try {
            mdEnc = MessageDigest.getInstance("MD5");
            mdEnc.update(pass.getBytes(), 0, pass.length());
            pass = new BigInteger(1, mdEnc.digest()).toString(16);
            while (pass.length() < 32) {
                pass = "0" + pass;
            }
            password = pass;
        } catch (NoSuchAlgorithmException e1) {
            e1.printStackTrace();
        }
        return password;
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

}
